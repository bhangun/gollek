package tech.kayys.wayang.tool.repository;

import io.quarkus.hibernate.reactive.panache.PanacheRepository;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.wayang.tool.entity.AuthProfile;

import java.util.List;

@ApplicationScoped
public class AuthProfileRepositoryImpl implements PanacheRepository<AuthProfile>, AuthProfileRepository {

    @Override
    public Uni<List<AuthProfile>> listAllProfiles() {
        return listAll();
    }

    @Override
    public Uni<List<AuthProfile>> findByRequestId(String requestId) {
        return list("requestId", requestId);
    }

    @Override
    public Uni<List<AuthProfile>> findByRequestIdAndEnabled(String requestId, boolean enabled) {
        return list("requestId = ?1 AND enabled = ?2", requestId, enabled);
    }

    @Override
    public Uni<AuthProfile> findById(String profileId) {
        return find("profileId", profileId).firstResult();
    }

    @Override
    public Uni<AuthProfile> findByRequestIdAndProfileId(String requestId, String profileId) {
        return find("requestId = ?1 AND profileId = ?2", requestId, profileId).firstResult();
    }

    @Override
    public Uni<AuthProfile> save(AuthProfile profile) {
        return persist(profile);
    }

    @Override
    public Uni<AuthProfile> update(AuthProfile profile) {
        return persist(profile);
    }

    @Override
    public Uni<Boolean> deleteById(String profileId) {
        return delete("profileId", profileId).map(deleted -> deleted > 0);
    }

    @Override
    public Uni<List<AuthProfile>> searchProfiles(String query, Object... params) {
        return list(query, params);
    }

    @Override
    public Uni<Long> count() {
        return PanacheRepository.super.count();
    }

    @Override
    public Uni<Long> countByRequestId(String requestId) {
        return count("requestId", requestId);
    }
}